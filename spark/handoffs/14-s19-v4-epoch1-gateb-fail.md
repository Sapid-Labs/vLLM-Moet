# s19 (2026-07-18) — Sparkulator v4 epoch 1: GATE B FAIL (live pos-0 0.733 vs bar 0.768)

> Previous: 13-s19-dynamic-k-null.md
> Design: `../SPARKULATOR-DESIGN.md` (§5 gates; GATE B bar frozen before training)

## STATUS

**v4 epoch-1 trained and live-probed same day. GATE B FAILED — the
pre-registered stop condition triggered. epoch-3 dspark K=3 remains champion.**
Awaiting Joe's call: close the speculator track per design §7, or overrule the
gate (e.g. epoch 2 / higher LR on the new columns). Resting §4b serve restored.

## DONE

- **Surgery + GATE A (PASS):** ckpt `~/dspark-distill-data/ckpt-v4-init`
  (worker) — fc 5→6×6144, zero-init new block, aux ids +[78]. Step-0 LR-0 on
  the 520-row weak valset: pos-0 **0.7724** / pos-1 0.6674 ≡ epoch-3's recorded
  0.773/0.669. Surgery byte-clean (63/63 other tensors identical).
- **Trainer mods** (pinned clone `spark-c84b:~/Dev/speculators-v05`, commits
  7c8f1d1 + b7868a5 on top of 21033a7): `DSPARK_V4_ALL_CHANNELS=1` env feeds
  ALL hs channels to the combine-fc in **ArrowDataset._get_raw_data** (the live
  path for safetensors caches) and in standardize_data_v1 (the .pt path);
  channel 6 stays the tv/ce label source. fc width auto-follows
  `len(aux_hidden_state_layer_ids)` in both trainer and fork — no other code.
- **Epoch 1:** full `prepared-combined-v3` (5200 rows), lr 1e-5, ~2.5 h
  (`~/dspark-train-v4.log`, ckpt `~/dspark-distill-data/ckpt-v4/0`, 11 GB with
  optimizer). Trainer val (tail 10% = weak band): pos-0 **0.773 — flat** vs
  epoch-3's 0.773. Expected (weak band = intrinsic entropy, s18). The new fc
  block DID learn: norm 31.8 (old block 394, drift 12.1), fully dense.
- **GATE B live probe (FAIL):** graft `~/models/hf/GLM-5.2-speculator.v4e1`
  (both nodes), §4b serve, warm 8, frozen sparkbench:
  **pos-0 0.733 / 24.06 tok/s** vs same-day epoch-3 baseline 0.742 / 24.40
  (bar: pos-0 ≥ 0.768). No category up; coherence smoke clean (17×23 ✓).
  Record: `../dspark-distill/tools/sparkbench-v4e1-s19.json`.

## READING THE RESULT

- Not a plumbing failure so far as measurable: serve loaded the 6-channel fc
  (would shape-error otherwise), aux-78 flows (s15 patch active), channel order
  trainer==fork (78 last), and the extraction that produced the training data
  ran through this same fork path. Trainer-space AND live-space both flat —
  unlike v2 (trainer up, live flat), there is no transfer gap to chase: after
  one epoch the final-state channel simply hasn't converted to acceptance.
- **Why slightly WORSE, not just flat:** (a) boot band — epoch-3 pos-0
  reproduces at 0.739–0.748 across restarts, so 0.733 is at the band's edge;
  (b) the real component is **fine-tune drift, reproducing v3's pattern**: one
  epoch on the magpie+weak mix moved the proven old weights (drift norm 12.1)
  and v3 landed at live pos-0 0.732 vs v4e1's 0.733 — same ~1 pt fine-tune tax,
  and the new channel paid no dividend to offset it.
- **Blind spot the gate leaves open:** the trainer VAL split is the weak band,
  where s18 proved NO draft (MTP included) can gain — flat there was expected
  and carries no information about the thesis. The conditioning bet's payoff,
  if any, lives on the reasoning/math/structured bands. Nothing measured today
  looked there in trainer space.
- Also open: 1 epoch @ lr 1e-5 grew the zero-init block to only ~8% relative
  magnitude — "undertrained pathway" is not excluded by today's data.

## NEXT (Joe decides) — options ranked by information-per-cost

1. **Reasoning-band trainer eval (no training, ~30–60 min GPU):** LR-0 of
   ckpt-v4/0 vs epoch-3 on `~/dspark-distill-data/prepared-reasoning-sub`
   (both with DSPARK_V4_ALL_CHANNELS set appropriately). If v4 lifts there in
   trainer space → thesis alive, problem is transfer/deployment; if flat →
   conditioning bet is dead with high confidence. THE cheap decisive probe.
2. **Freeze-old / train-new-only at higher LR (~2–3 h):** param-group freeze on
   all epoch-3 weights, 10× LR on the new fc columns only. Kills the drift
   confound AND the undertraining objection in one run. Pre-commit a bar first.
3. Plain epoch 2 — weakest (v3 precedent: drift compounds).
4. +2 on-policy data rounds (~2 serve-days) — design §5.1 pre-committed this
   "before concluding anything", but trainer-space flatness lowers its prior.
5. Fork-native training (v3 Phase 5) — no mismatch signature to chase here;
   low prior, high effort.
6. **Close per §7** — keep epoch-3 (which already beats native MTP here).

## HOW TO RESUME / RE-RUN

- Train: `bash ~/launch_v4_epoch1.sh` on worker (edit --from-pretrained to
  ckpt-v4/0 + drop --no-resume-from-checkpoint for epoch 2). Serve must be
  DOWN (GPU). `DSPARK_V4_ALL_CHANNELS=1` is REQUIRED with 6-channel ckpts.
- GATE A/B protocol + exact commands: handoff 13 §HOW-TO-RESUME + this file's
  graft path. Live probe: sparkbench.py --warmup 8 vs a same-session epoch-3 run.

## GOTCHAS

1. `pgrep -f`/`pkill -f` self-match struck FOUR times this session (a stuck
   GATE A launcher, a phantom "training running", a killed ssh shell): any
   pattern that appears in your own bash -c/ssh command string matches the
   wrapper. Check processes with `ps aux | grep "[s]cripts/train.py"` (bracket
   trick) and kill by PID. The epoch-1 launch lost ~30 min to this.
2. Trainer requires `--train-data-ratio` strictly < 1 — use 0.999 for
   gate-style full-set passes.
3. GATE-style runs write full 11 GB checkpoints (model + optimizer) even at
   LR 0 — save to /tmp and delete, or worker disk vanishes.
4. Worker disk after this session: ~15 GB free. Reclaim candidates, in order:
   `ckpt-v4/0` optimizer_state_dict.pt (~4 GB, only if not doing epoch 2),
   `~/models/hf/GLM-5.2-speculator.v3` (7.1 GB per node, s18-designated),
   ckpt-v3 (7.1 GB). KEEP both hs caches + ckpt-v4-init + v4e1 graft
  (hardlinked, ~0 extra) until the v4 decision is made.

## LINKS

- Records: `../dspark-distill/tools/sparkbench-v4e1-s19.json` (+ tau0 baseline
  from handoff 13)
- Trainer clone commits: 7c8f1d1, b7868a5 (worker, `~/Dev/speculators-v05`)
- Session tracker: `/sessions/sparkulator-dynamic-k.json`
