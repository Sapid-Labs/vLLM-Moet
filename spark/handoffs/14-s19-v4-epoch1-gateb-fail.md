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
- Open question the gate leaves unanswered: whether 1 epoch @ lr 1e-5 is just
  too gentle for a zero-init pathway (the block grew to only ~8% relative
  magnitude), vs the bet itself being wrong (parallel drafting caps acceptance
  regardless of conditioning). The frozen protocol deliberately does not let us
  spend more compute to find out without an explicit decision.

## NEXT (Joe decides)

1. **Close per §7** — keep epoch-3, done; or
2. **Overrule GATE B once**: epoch 2 (resume from ckpt-v4/0, same data) and/or
   a targeted LR bump on the new fc columns; re-probe live. If overruling,
   pre-commit the new bar first.

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
