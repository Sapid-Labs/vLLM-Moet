# HANDOFF — DSpark draft self-distillation (squeeze GLM-5.2 dspark acceptance)

Resume-here pointer for the dspark draft fine-tune. Deep docs alongside:
`PLAN.md` (design + decisions), `FINDINGS.md` (full investigation + rule-outs).
Convention: `~/CLAUDE.md` → "Session handoffs".

## STATUS (2026-07-17 late, session 18) — v3 VERDICT: NULL (third round). Weak-band content has NO learnable signal even in trainer space. Phase 5 NOT auto-triggered (its condition didn't occur) — awaiting Joe's call.

Round 3 ran end-to-end in one day (phases 0-4 below all DONE). Results:
- **Live back-to-back A/B on frozen sparkbench (same evening, clean-restart protocol,
  warm 8):** epoch-3 **23.15 tok/s, pos-0 0.748, tok/verify 2.28** vs v3 **22.70,
  0.732, 2.241** → v3 is a small consistent LOSS. (epoch-3 reproduces across boots:
  0.743 morning / 0.748 evening — the protocol works; per-category numbers on n=4
  prompts swing ±5-8pt across boots, only the 52-prompt mean is meaningful.)
  Records: `tools/sparkbench-{epoch3-baseline,epoch3-ab,v3}-s18.json`.
- **Trainer-space LR-0 diagnostic on an identical 520-row weak valset**
  (`~/dspark-distill-data/prepared-weak-valset`, rows 4680-5199 of combined-v3):
  epoch-3 **0.773/0.669** (pos-0/1) vs ckpt-v3 **0.778/0.680** → **+0.5pt** — the
  fine-tune barely moved the trainer's OWN metric on the weak band, despite
  on-policy data, correct masks (0.797), healthy trainer (GATE-1 0.832).
- **⇒ This is the "val flat too" branch of the plan**: the weak band (creative/
  chat/summarize prose) is intrinsic content entropy, not a trainable gap. Unlike
  v2 (val +2.4pt, live flat = transfer gap), v3 shows there is NO SIGNAL to
  transfer. Fork-native training (Phase 5) fixes metric transfer — it cannot
  create signal. Its pre-approval condition (val-up-live-flat) did NOT occur.
- **Three-round convergence:** v1 (familiar content) null, v2 (no-headroom
  reasoning) null, v3 (the actual weak distribution, on-policy) null-to-negative.
  **epoch-3 is at the practical ceiling of this draft on this target.**
- Artifacts kept: `ckpt-v3` (worker, optimizer states deleted from v1/v2/v3 for
  disk), `~/models/hf/GLM-5.2-speculator.v3` (both nodes, 7.6 GB each —
  reclaimable), `prepared-weak-sub`/`prepared-combined-v3`/`prepared-weak-valset`,
  `~/dspark-hs-weak` (90 GB, worker — the big reclaim candidate if wound down).
  **Worker disk is at ~12 GB free — reclaim before any new work.**
- Resting state: epoch-3 dspark K=2 §4b serving on :8000 (`~/serve-dspark-best.log`).

## STATUS (2026-07-17, session 18 morning) — EFFORT REOPENED: Sparkulator v3 (Joe's call — goal: BEAT epoch-3 on overall sparkbench mean). Phase 1 (weak-band gen) RUNNING.

Full plan: `~/.claude/plans/dazzling-seeking-steele.md` (approved). Win condition =
beat stock epoch-3 on **mean tok/s AND tok/verify across the frozen 52-prompt
sparkbench**, back-to-back live. If val-up-live-flat repeats → fork-native training
is pre-approved (train through the fork's own forward).

- **Tooling rescued into the repo** (was in volatile /tmp): `tools/` now has
  probe_weak_content.py, gate3_measure.py (KNOWN-BUGGY measurement — superseded),
  gen_completions.py, build_reasoning_prompts.py, combine_datasets.py,
  measure_decode.py, **sparkbench.py** (the fixed, canonical A/B tool:
  usage-token decode rate + per-prompt /metrics deltas over num_drafts_total)
  and **sparkbench-prompts.jsonl** (frozen: 40 probe + 12 held-out reasoning).
- **Epoch-3 baseline (healthy machine, s18)** — `tools/sparkbench-epoch3-baseline-s18.json`:
  overall **23.13 tok/s, pos-0 0.743, pos-1 0.523, tok/verify 2.266**. Weak band
  confirmed: code_explain 0.662, creative 0.683 (pos-1 0.359!), summarize 0.706,
  code_gen 0.708. Strong: math 0.875, logic 0.777, heldout_reasoning 0.763.
  (Session-16 absolute tok/s numbers were environmental garbage — see
  `../handoffs/04-decode-throughput-regression.md` RESOLUTION. Protocol now:
  fresh cluster+serve, warm ≥8, same-session A/B only.)
- **Weak-band dataset built** (`tools/build_weak_prompts.py`, seed 0):
  `~/dspark-distill-data/weak-prompts-train.jsonl` (4320: creative 1080,
  chat_open 900, summarize 720, code_explain 720, code_gen 540, structured 360)
  + `weak-prompts-holdout.jsonl` (480, never trained, for the final A/B).
- **Phase 1 DONE (2026-07-17 eve)**: 4320/4320 greedy completions →
  `~/dspark-distill-data/weak-completions.jsonl` (one mid-run engine crash —
  `RPC call to sample_tokens timed out` after ~3.3 h sustained batch — resumed at
  concurrency 12 after a fresh cluster restart, 0 errors). prepare_data with the
  explicit GLM assistant pattern: **loss_mask 0.796/0.797 mean, 0 empty** (gate PASS).
  Stratified 3200-row subsample → `prepared-weak-sub` (on both nodes).
- **Phase 2 DONE**: hidden states extracted on the fast-build hidden serve
  (3199/3200, sample 0 skipped, ~50 min at concurrency 4) and drained to
  `spark-c84b:~/dspark-hs-weak` (90 GB; worker now ~30 GB free — tight, watch it).
  Combined dataset assembled on the worker: `prepared-combined-v3` = 5200 rows
  (magpie 2000 + weak 3200), 5187 hs links, max index check OK
  (`tools/combine_datasets_v3.py`).
- **Phase 3 RUNNING**: GATE-1 step-0 LR-0 **PASS (pos-0 epoch 0.832** vs broken 0.35);
  fine-tune v3 launched on the worker (`~/dspark-finetune-v3.log`, pinned v05 trainer,
  from epoch-3, lr 1e-5, 1 epoch, save `~/dspark-distill-data/ckpt-v3`). ETA ~2-6 h.
- NEXT after train: GATE-2 coherence spot-check, graft ckpt-v3/0 weights into a clone
  of the reference speculator dir, serve §4b with `SPECULATOR=<clone>`, live-probe with
  `tools/sparkbench.py` (NEVER select by trainer val), then Phase-4 back-to-back A/B
  vs epoch-3 under the clean-restart protocol. Weak-category targets: creative
  0.683→≥0.72, code_explain 0.662→≥0.70.

## STATUS (2026-07-17, session 16g) — EFFORT CONCLUDED: epoch-3 near ceiling; dspark-distill wound down (Joe's call) — **REOPENED in s18 above**

Ran an empirical weak-content probe on epoch-3 (40 prompts, 10 categories, per-prompt
/metrics acceptance). Result — pos-0 by category, weakest→strongest:
creative_writing 0.675 · code_explain 0.713 · code_gen 0.718 · chat_open 0.720 ·
summarize 0.721 · science_deriv 0.722 · structured 0.746 · factual 0.756 ·
logic_reasoning 0.778 · math_derivation 0.909. Overall mean 0.746, spread 0.388.
KEY: the weak content is HIGH-ENTROPY creative/open-ended PROSE, not reasoning/code
(which sit 0.72-0.91, near ceiling). v2 trained on the STRONG end → explains its null.
The weak end is (a) likely intrinsic entropy, not a trainable gap, and (b) low-value to
accelerate. **Joe's decision: STOP — epoch-3 is the shipped draft; do not publish
Sparkulator.** Probe data: `~/weak-content-probe.json`, script `scratchpad/probe_weak_content.py`.

THROUGHPUT FOLLOW-UP (separate from the verdict): the GATE-3 serves measured ~10.4 tok/s
single-stream vs the documented 21.8 — for BOTH v2 and epoch-3 equally. Ruled out:
v2-specific (weights byte-near-identical), page cache (fadvise-DONTNEED evict of 300 GB
hidden states did nothing), PLANES_MMAP (hardcoded on). Most likely THERMAL THROTTLE after
many hours of sustained GB10 load (generate→extract→train→serve); should recover after
cooldown. Re-verify with a fresh serve later; no clean pre-session baseline was captured.

RECLAIMABLE DISK (worker, if not resuming): `~/dspark-hs-reasoning-in` (~85 GB),
`~/dspark-distill-data/prepared-combined/hidden_states` (symlinks), `~/dspark-hs-staging`,
`~/dspark-hs-reasoning-out`, head `~/dspark-distill-data/openhermes-src` (1.9 GB). KEPT:
ckpt-v1, ckpt-v2, prepared-reasoning*, reasoning-completions.jsonl. Current serve: epoch-3
dspark K=2 §4b on :8000 (baseline restored, matches session start).

## STATUS (2026-07-17, session 16f) — GATE 3 VERDICT: v2 = NULL (no live-proposer gain); back-to-back A/B corrected the false win

FINAL GATE-3 result, same-instance back-to-back on 12 held-out reasoning prompts
(~2830 draft steps each, cumulative /metrics per_pos_total/num_drafts_total):
| draft | pos-0 | pos-1 | tok/verify | decode tok/s |
| v2      | 0.771 | 0.561 | 2.33  | 10.19 |
| epoch-3 | 0.770 | 0.556 | 2.325 | 10.22 |
**IDENTICAL → the reasoning fine-tune produced NO live-proposer acceptance gain.**
The earlier "+9pt win" was an ARTIFACT of comparing v2 (0.771) to the DOCUMENTED 0.68
baseline, which was measured on a DIFFERENT (harder) ablation prompt set — not
apples-to-apples. Lesson: always A/B the two drafts back-to-back on the SAME prompts
in the SAME session; never compare to a doc number from another content set.

WHY NULL (diagnosis): (1) the held-out OpenHermes code/derivation/reasoning prompts
already score 0.77 under epoch-3 — they were NOT the weak content; dspark's 0.68
weakness lives in the specific ablation "300-tok derivation" distribution, which my
OpenHermes slice did not match. Trained where there was no headroom → no gain.
(2) trainer≠proposer: val moved to 0.813 in the TRAINER's metric space but the fork
proposer didn't budge. Weights DID change (attn proj 1-4%, verified), so not a save bug.
Absolute 10.2 tok/s (both) = environmental page-cache thrash (300 GB hidden states
squatting), NOT v2-specific — affects both equally; not the deciding factor.

**DO NOT publish Sparkulator — no gain to ship.** Paths if resuming: (a) get/reconstruct
the ACTUAL ablation weak-content prompt set and train on THAT distribution; (b) higher
LR / more epochs for a bigger weight delta (risk: degrade); (c) accept the fork
proposer has a ceiling this offline-distill approach can't move and stop. Artifacts kept:
ckpt-v2 at `~/dspark-distill-data/ckpt-v2/0` + `~/models/hf/GLM-5.2-speculator.v2` (both
nodes); reasoning data `~/dspark-distill-data/prepared-combined` + hidden states.

## STATUS (2026-07-17, session 16e) — GATE 3: v2 ACCEPTANCE WIN (77/56 vs 68/45); throughput anomaly under A/B check

**v2 dspark K=2 live on 12 HELD-OUT reasoning prompts: pos-0 0.771, pos-1 0.561,
tok/verify 2.33** (from /metrics per_pos_total/num_drafts). Epoch-3 baseline was
0.68/0.45, 2.14. → **+9pt pos-0, +11pt pos-1, +0.19 tok/verify** — the reasoning
slice WORKED, acceptance improved on the weak domain, on held-out prompts.
CAVEAT — absolute decode measured **10.2 tok/s** (flat over 16+ warm runs, below the
15.6 no-draft floor), i.e. ~2× slower than the 21.8 baseline. NOT v2-specific: v2 and
epoch-3 speculator config.py + weight keys/shapes/dtypes are BYTE-IDENTICAL (only a
cosmetic transformers_version string differs), so draft fwd cost is identical → the
slowdown is ENVIRONMENTAL (suspect: ~300 GB of freshly-written hidden states squatting
in page cache, evicting plane pages; RAM shows 88-93 GB buff/cache but planes may be
thrashing). At equal step cost, v2's higher tok/verify makes it strictly ≥ epoch-3 in
tok/s. Running back-to-back epoch-3 on the SAME instance (`~/serve-ep3-gate3.log`) to
prove it + quantify the environmental factor. Measure script `scratchpad/gate3_measure.py`
(note: its tok_per_verify field uses draft-TOKENS denom — WRONG; use /metrics
per_pos_total/num_drafts_total for the real per-pos rate, as above).
TODO after A/B: fix throughput (drop page cache / re-warm; may need sudo drop_caches),
re-measure v2 warm for the true tok/s, then if win holds → publish Sparkulator.

## STATUS (2026-07-17, session 16d) — ckpt-v2 DONE (val reasoning pos-0 0.813, +); GATE 3 serve RUNNING

ckpt-v2 completed in ~2 h (faster than the 6 h estimate). **VAL on the reasoning-heavy
split: position_1_acc 0.813, position_2 0.724, full_acc 0.612** — above the old magpie
baseline 0.789, and reasoning batches climbed ~0.75→0.813 within the epoch. Positive
leading indicator that the reasoning slice moved acceptance on the weak domain. ckpt at
`~/dspark-distill-data/ckpt-v2/0` (reference layout, arch DSparkDraftModel), staged on
BOTH nodes as `~/models/hf/GLM-5.2-speculator.v2`. GATE 3 serve UP: dspark K=2 §4b with
`SPECULATOR=~/models/hf/GLM-5.2-speculator.v2` (`~/serve-v2-gate3.log`). Measuring decode
tok/s + per-pos acceptance on 12 HELD-OUT reasoning prompts (rows 3200+, not trained)
via `scratchpad/gate3_measure.py` (streaming, /metrics deltas). Baseline to beat: epoch-3
dspark K=2 = 68/45% pos-0/1, ~21.8 tok/s. Plan: measure v2, then back-to-back swap
SPECULATOR to epoch-3 (`GLM-5.2-speculator.dspark`) on identical prompts for a clean
same-session A/B. If v2 wins significantly → publish "Sparkulator" (sapidlabs).

## STATUS (2026-07-17, session 16c) — step-3 data pipeline DONE; ckpt-v2 fine-tune RUNNING (~6 h)

Full reasoning-slice pipeline completed: 4800 on-policy completions → prepare_data
(w/ the mask-bug fix) → 3200-sample subsample → 3197 hidden states extracted +
drained to worker → combined dataset assembled (`~/dspark-distill-data/prepared-combined`,
5200 rows = 2000 magpie + 3200 reasoning, 5185 hs symlinks, index-aligned, verified).
**ckpt-v2 fine-tune RUNNING** on the worker (v05 venv confirmed): `--data-path
prepared-combined --from-pretrained epoch-3 --lr 1e-5 --scheduler-type none --epochs 1`,
save → `~/dspark-distill-data/ckpt-v2`, log `~/dspark-finetune-v2.log`. ~181 tok/s over
4.08M tok → ~6 h. Early: magpie batches pos-0 ~0.84, reasoning batches lower (~0.50
full_acc) = the ones we want to lift. VAL split is the tail 10% = reasoning-heavy, so
`val/position_1_acc_epoch` is the KEY signal (v1's magpie-val was 0.785 flat). Waiter
`bor0ujiqk` fires on completion. NEXT: GATE 2 coherence, then GATE 3 — deploy ckpt-v2/0
(reference layout, no grafting), serve dspark K=2 §4b, warm ≥10, measure pos-0/1 accept
+ tok/s vs 21.8/68%. If significant gain → publish as "Sparkulator" (sapidlabs HF).

## STATUS (2026-07-16, session 16b) — v1 DONE but NULL (as predicted); step 3 (reasoning slice) IN PROGRESS

v1 fine-tune completed cleanly (~1.5 h, ckpt at `spark-c84b:~/dspark-distill-data/ckpt-v1/0`,
saved in the EXACT reference layout: architectures `["DSparkDraftModel"]` + auto_map
+ config.py + 7.6 GB model.safetensors → **deployable as-is, no grafting needed**).
But val pos-0 = **0.785 ≈ 0.789 baseline → no acceptance gain**, exactly as PLAN
predicted for magpie/ultrachat-only content. Joe chose to skip a confirmatory GATE-3
serve of v1 and go straight to step 3 (the real lever).

**Step 3 = on-policy reasoning slice** (Joe's domain pick: technical explanation/
derivation + code/code-explanation + general reasoning; NO math). Pipeline:
1. DONE — prompts at `~/dspark-distill-data/reasoning-prompts.jsonl` (4800, 1600
   each code/derivation/reasoning; extract script `scratchpad/build_reasoning_prompts.py`,
   filters OpenHermes-2.5 `source` tags, excludes math). Pool was deep (77k-180k/domain).
2. IN PROGRESS — generation via `scratchpad/gen_completions.py` (stdlib, ThreadPoolExecutor
   c=16, temp0, max_tokens 300, resumable) → `~/dspark-distill-data/reasoning-completions.jsonl`.
   Serve = throughput-tuned: `serve-glm52-tp2-dspark.sh --no-spec --max-num-seqs 16
   --max-num-batched-tokens 2048 --kv-cache-memory-bytes 6G --hf-overrides topk4`
   (`~/serve-gen-throughput.log`), env same as §4b (nvfp4_big3_overlay + p208_reap planes).
   Instantaneous ~72 tok/s aggregate (MoE bandwidth-bound; batching sublinear) → ETA ~5 h
   for full 4800. Nearly all completions hit the 300-tok length cap. Resumable + incremental
   so can harvest a balanced subset early. **Tear down this serve before the extract stage.**
   DONE: 4800/4800, 0 err, ~1.25M tok, balanced. Coherent (target's "planning" style).
3. DONE — `prepare_data.py` → `~/dspark-distill-data/prepared-reasoning` (4800 rows).
   **CRITICAL GOTCHA — assistant mask**: GLM-5.2's chat template has NO `{% generation %}`
   marker, so `return_assistant_tokens_mask` returns ALL ZEROS, and the auto-detected
   regex `<|assistant|>(...)<|user|>` needs a trailing `<|user|>` that SINGLE-turn convos
   lack → **loss_mask 100% empty** (silent; training learns nothing). Fix = explicit
   `--assistant-pattern '<\|assistant\|>(.*?)(?=<\|user\|>|<\|assistant\|>|$)'` → mean
   loss_mask 0.81, zero empty (matches magpie 0.83). ALWAYS verify loss_mask>0 for GLM.
4. Extract (IN PROGRESS). **DISK CONSTRAINT**: full 4800 = 137 GB bf16; worker only 127 GB
   free, head 87 GB, fp8 won't work (magpie is bf16, must match). → SUBSAMPLED to
   `prepared-reasoning-sub` (first 3200, balanced, 1.24M tok, ~91 GB) = ~30% token-mix.
   `serve-glm52-tp2-hidden.sh` (fast-build env + top-k4, HIDDEN_STATES_PATH=~/dspark-hs-staging)
   + `data_generation_offline.py --output ~/dspark-hs-reasoning-out` → `hs_{idx}`, rsync
   `--remove-source-files` drain head→worker DURING gen. hs is BF16 [seq,6,6144]+I64 token_ids.
   MIXING: trainer matches `hs_{row_index}`; combined = concat(magpie 2000, reasoning-sub
   3200) + hs = magpie hs_0..1999 + reasoning hs_i→hs_{2000+i} (magpie 2000 rows/1988 hs,
   12 gaps ok via --on-missing skip).
5. Mix reasoning slice + existing magpie prepared cache → fine-tune (v05 trainer on
   worker) → `ckpt-v2`. GATE 2 coherence, then GATE 3 A/B vs 21.8/68% baseline.

VENV DISCIPLINE: data-prep/gen scripts live in `~/Dev/speculators` (HEAD) and need the
HEAD package → run them ON THE HEAD NODE (head venv = HEAD 0.7.0.dev102). The trainer
needs v05 → runs on the WORKER (worker venv currently = v05 0.7.0.dev74). No swap needed
if each stage runs on its correct node.

## STATUS (2026-07-16, session 16) — GATE 1 PASSED (version pin fixes it); fine-tune v1 RUNNING on the worker

Session 15's blocker is RESOLVED. Pinned the trainer to `21033a7` in a separate
clone `spark-c84b:~/Dev/speculators-v05` (`pip install --no-deps -e .` into the
`vllm-moet` venv — this SHADOWS the HEAD install; reinstall HEAD
`pip install --no-deps -e ~/Dev/speculators` before any extract/gen). Ran the
STEP-0 LR-0 gate on the FP8 cache: **mean position_1_acc (card "pos-0") = 0.789
across 8 real steps** (0.788/0.835/0.780/0.875/0.713/0.667… noisy on 30 rows),
vs the broken **0.35** at HEAD. GATE 1 PASS confirmed — the root cause was
exactly the 07-13 speculators forward rewrites; nothing wrong with the pipeline.
NOTE the trainer positions are **1-indexed** (position_1 = card pos-0).

**Fine-tune v1 launched** (NEXT step 2): full 1988-cache, `--from-pretrained`
epoch-3, `--lr 1e-5 --scheduler-type none --epochs 1`, Muon (lr 1e-4) + AdamW
(1e-5), loss ce0.1/tv0.9, save → `spark-c84b:~/dspark-distill-data/ckpt-v1`,
log `~/dspark-finetune-v1.log`. Early steps already tick up (step 15:
position_1_acc 0.818, position_2_acc 0.729). ~8.5 s/step, ~251 tok/s →
**ETA ~3-3.5 h for 1 epoch** (~3M tok). A background waiter fires on
save/crash. We did NOT re-benchmark the epoch-3 baseline first (Joe's call):
the version bug was purely in TRAINING, so re-serving the unchanged draft would
just reproduce the documented 21.8 tok/s / 68%. A/B happens AFTER, vs that
recorded baseline.

**NEXT after v1 finishes:** GATE 2 greedy coherence spot-check, then GATE 3 —
convert/point `--speculative-config` at ckpt-v1, serve dspark K=2 (§4b), warm
≥10, read pos-0/1 acceptance from /metrics + tok/s via `spark/demo_chat.py`.
Then step 3 (reasoning-heavy slice) is the real lever. Both GPUs are needed for
the TP2 serve, so the fine-tune must be DONE (or killed) before A/B.

**GATE-3 deploy recipe (pre-scouted session 16):** the fork serve loads the
reference speculator `~/models/hf/GLM-5.2-speculator.dspark`, whose `config.json`
has `architectures:["DSparkDraftModel"]` + `auto_map`→`config.DSparkSpeculatorConfig`
+ a bundled `config.py` + `model.safetensors` (7.6 GB). The trainer (pinned
21033a7) saves via `draft_model.save_pretrained(save_path)`, which may emit a
DRIFTED config (older speculators — verify architectures + auto_map + that
config.py is present). ROBUST path: `cp -r` the reference dir → clone, overwrite
its `model.safetensors` with `ckpt-v1/model.safetensors` IFF tensor keys match
(check with safetensors header diff), keep the reference config.json/config.py.
Point `--speculative-config`'s `model` at the clone. This grafts new weights
into the proven layout and sidesteps config-format drift entirely.

## STATUS (2026-07-16, session 15) — pipeline BUILT + root-caused; blocked on speculators version drift; next = pin the trainer version

Goal: raise the dspark draft's acceptance on the shipped fast-build target
(dspark K=2 + REAP p208 + NVFP4 big-3 + top-k4, ~21.8 tok/s) from measured
pos-0/1 ≈ 68/45% toward the card's 83/72%, for an estimated +2-4 tok/s
(→ ~24-25 ceiling at K=2-3). The whole offline pipeline works; it is blocked
by ONE thing: the trainer (speculators repo HEAD) is a drifted implementation
of the dspark forward vs the checkpoint's training version, so a correctly
loaded checkpoint scores pos-0 0.35 at step-0 instead of the card's 0.83.
**Fix = run the trainer at the checkpoint's era (~commit 21033a7), not HEAD.**

## DONE (verified)
- **Hidden-states extraction server on the fork**: `spark/serve-glm52-tp2-hidden.sh`
  (extract_hidden_states + ExampleHiddenStatesConnector, chunked-prefill off,
  eager). Needs TWO venv patches, tracked in `patches/`, applied+synced BOTH nodes:
  1. `deepseek_v2.py`: emit pre-norm final hidden when aux id == num_layers (78),
     so the trainer gets the last-layer state for tv/ce targets.
  2. `example_hidden_states_connector.py`: `.pop(req_id, None)` — aborted requests
     (client timeouts) otherwise KeyError-kill the EngineCore.
- **Cache generated**: 1988 samples (~3M tok, ~205 GB) at
  `spark-c84b:~/dspark-distill-data/prepared/hidden_states/hs_<idx>.safetensors`,
  format `{hidden_states:[seq,6,6144], token_ids:[seq]}` (5 aux layers
  [8,23,39,55,70] + last layer 78). Prepared arrow dataset (2000 magpie/ultrachat
  samples, GLM assistant-pattern) at `~/dspark-distill-data/prepared` on both nodes.
- **Trainer runs** on the worker: `--from-pretrained` epoch-3, single-GPU bf16,
  Muon, sdpa/flex both work, ~100 tok/s, mem < 100 GB at max-anchors 1024.
- **Root cause found — all alternatives RULED OUT** (see FINDINGS.md): draft
  weights byte-match checkpoint; tv/ce targets correct (0.607 = card 0.613); aux
  collection point correct (live-proposer fc-input diff vs cache = fp8 noise only,
  magnitudes match); loss_mask correct (81%, clean split); attn impl irrelevant.
  The ONLY thing left is the speculators forward rewrite between the checkpoint's
  `0.5.0.dev38` and installed HEAD `0.7.0.dev102` (`sample_from_anchor` #760,
  default sliding-window #749, `create_block_mask` compile #731, argmax cleanup
  #730 — ALL dated 2026-07-13, after the checkpoint was trained).
- **Version target identified**: `21033a7` (2026-07-08) — after dspark-add
  (ff71b1e, 06-29), before the 07-13 forward rewrites; has offline gen +
  --from-pretrained + dspark core; lacks sample_from_anchor. Statically verified.

## NEXT (the concrete plan — do in order)
1. **Pin the trainer version + confirm step-0.** In a SEPARATE clone (don't
   disturb the HEAD install the extract server's tooling uses — though the serve
   itself is vLLM, not speculators):
   ```
   git clone ~/Dev/speculators ~/Dev/speculators-v05 && cd ~/Dev/speculators-v05
   git checkout 21033a7
   ~/venvs/vllm-moet/bin/pip install --no-deps -e .   # NOTE: breaks HEAD install;
   #   reinstall HEAD (`pip install --no-deps -e ~/Dev/speculators`) before using
   #   the extract/gen scripts again, OR use a second venv for the trainer.
   ```
   Then the STEP-0 GATE on the worker (serve must be DOWN — needs the GPU):
   ```
   ssh spark-c84b '~/venvs/vllm-moet/bin/python ~/Dev/speculators-v05/scripts/train.py \
     --verifier-name-or-path ~/models/hf/GLM-5.2-FP8 --speculator-type dspark \
     --from-pretrained ~/models/hf/GLM-5.2-speculator.dspark \
     --data-path ~/dspark-distill-data/prepared-mini \
     --hidden-states-path ~/dspark-hs-fp8test \
     --save-path /tmp/t --epochs 1 --lr 0 --scheduler-type none \
     --total-seq-len 4096 --max-anchors 1024 --draft-attn-impl sdpa \
     --loss-fn "{\"ce\":0.1,\"tv\":0.9}" --on-missing skip --log-freq 1 \
     --num-workers 2 --trust-remote-code'
   ```
   PASS = pos_0_acc ≈ 0.80-0.83 (vs the broken 0.35 at HEAD). If still 0.35,
   BISECT `ff71b1e..a0be7bb` (git bisect on the step-0 metric); the offending
   commit is likely a0be7bb/4fd632c/f9e9685 (all 07-13). Do NOT proceed to a real
   fine-tune until this gate passes — a fine-tune from a broken baseline is worthless.
2. **Fine-tune on the EXISTING cache** (fast-build hidden states, magpie/ultrachat
   content): same command, drop `--hidden-states-path`/`--lr 0`, use
   `--data-path ~/dspark-distill-data/prepared` (the 1988-cache), `--lr 1e-5`,
   `--epochs 1-2`, `--save-path ~/dspark-distill-data/ckpt-v1`. This is the
   pure "align to fast-build target" run. Expected: small gain (target-alignment
   was measured to cost ≤1pt, so this mostly re-teaches familiar content).
3. **Generate the reasoning-heavy slice** (the real lever — task #7). Use the
   fast-build target's OWN greedy outputs on reasoning/technical prompts (the
   content where dspark is weak — the 300-tok explanation/derivation style from
   the ablation prompt set). Serve fast-build normally, generate ~1-2M tok of
   completions, format as jsonl conversations, `prepare_data.py`, then the
   hidden-states server + offline gen. Mix ~30-50% with magpie. Re-fine-tune.
4. **Deploy + A/B.** Point `--speculative-config`'s `model` at the new checkpoint
   dir (may need conversion to the speculators HF layout the fork's loader
   expects — verify `SpeculatorsConfig.from_pretrained(ckpt).architectures ==
   ['Qwen3DSparkModel']`). Serve dspark K=2, warm ≥10 runs, read pos-0/1
   acceptance from /metrics, measure tok/s with `spark/demo_chat.py`. Re-sweep
   K=2 vs K=3 (higher acceptance may re-open K=3).

## HOW TO RESUME
- **Baseline serve (the thing to beat, also the current live server)**: §4b of
  `spark/RUNBOOK.md` — dspark K=2 + REAP + NVFP4 + top-k4 ≈ 21.8 tok/s.
  `~/serve-dspark-best.log`. Tear it down before any trainer/gen run (GPU).
- **Extract server** (for new hidden states): `HIDDEN_STATES_PATH=<dir>
  MODEL=$M/nvfp4_big3_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208_reap
  VLLM_NVFP4_DENSE=1 VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj
  VLLM_MOE_W2_FADVISE_GLOB= bash spark/serve-glm52-tp2-hidden.sh
  --hf-overrides '{"num_experts_per_tok":4}'` (add nothing for plain-FP8).
  WARM with 2 completions before generating (cold prefill > 120s default).
- **Offline gen**: `~/Dev/speculators*/scripts/data_generation_offline.py
  --model glm-5.2 --endpoint http://localhost:8000/v1 --preprocessed-data
  <prepared> --output <hs-dir> --concurrency 4 --request-timeout 900`. Drain to
  worker with an rsync `--remove-source-files --min-size=1` loop (head disk ~97 GB).
- **Step-0 eval harness**: `~/dspark-distill-data/prepared-mini` (30 rows) +
  `~/dspark-hs-fp8test` (FP8 hidden states) + the train.py LR-0 command above.
- Speculator ckpt on BOTH nodes: `~/models/hf/GLM-5.2-speculator.dspark` (7.6 GB).

## GOTCHAS / KEY FACTS
- **speculators version tags are NOT chronological** (v0.6.0 dated 06-12,
  v0.5.0 older, but dspark-add ff71b1e is 06-29 AFTER both). Trust commit DATES
  and the step-0 gate, not the version string. Checkpoint self-reports 0.5.0.dev38.
- **The trainer forward ≠ the fork's inference proposer** are two codebases;
  BUT the original 0.5.x→fork pipeline empirically works (the shipped draft gets
  68% under the fork proposer), so training at the checkpoint's era and deploying
  on the fork is a proven recipe — the (b)≠(c) worry in FINDINGS is not blocking.
- **Ray does NOT forward driver env to workers** — the proposer/model runner runs
  on workers; env-gated debug (VLLM_DUMP_AUX etc.) must be hardcoded, not exported.
- **`--model` to the gen script is an equality check** vs the server's model list;
  pass the SERVED alias `glm-5.2`, not the path.
- **Cold page cache**: first ~10 runs after any serve boot are slow (planes fault
  in via mmap); never benchmark cold. Warm the extract server too before gen.
- **kill self-match**: `pgrep -f 'train.py'|kill` and `pkill -f 'vllm serve'`
  match their own launcher shell → exit 144. Use bracket class `[t]rain.py`,
  `[b]in/vllm serve`, and kill by PID.
- **Trainer is single-GPU** on the worker; the head can serve meanwhile IF the
  trainer only uses the worker — but TP2 serve needs BOTH GPUs, so serve must be
  fully down for any worker-GPU trainer/gen run.
- **`--scheduler-type` choices are linear|cosine|none** (NOT "constant").
- Aux last-layer patch: this fork collects aux at LAYER ENTRY inside the loop
  (aux id i = output of layer i-1); id==num_layers(78) is handled after the loop
  (pre-norm final). Verified correct via lm_head→0.607.

## CORRECTNESS GATES
- **GATE 1 (blocks everything)**: step-0 LR-0 on the version-pinned trainer must
  give pos_0_acc ≈ 0.80-0.83 on the FP8 cache. Until then the pipeline is wrong.
- **GATE 2**: after fine-tune, greedy coherence spot-check (target quality is
  unchanged by the draft — no battery needed).
- **GATE 3 (ship)**: dspark K=2 pos-0 acceptance ≥ ~75% AND ≥ 23 tok/s steady on
  the ablation prompt set, else keep epoch-3 and report honestly.

## LINKS
- `PLAN.md` (design), `FINDINGS.md` (investigation), `patches/` (2 venv patches).
- Ablation that motivates this: `spark/handoffs/03-dspark-acceptance-ablation.md`.
- Serve recipes: `spark/RUNBOOK.md` §4b (fast build) / §4c (this effort).
- Speculator: https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark
- Results log: `~/Dev/howtospark/models/glm-5.2.md`.
